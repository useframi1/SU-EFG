import {
  Component,
  ElementRef,
  inject,
  OnInit,
  ViewChild,
} from '@angular/core';
import { Prompt } from '../_models/prompt';
import { Message } from '../_models/message';
import { ChatbotService } from '../_services/chatbot.service';
import { MessageBoxComponent } from '../_components/message-box/message-box.component';
import { PromptFormComponent } from '../_components/prompt-form/prompt-form.component';
import { RouterLink } from '@angular/router';

@Component({
  selector: 'app-home',
  standalone: true,
  imports: [MessageBoxComponent, PromptFormComponent, RouterLink],
  templateUrl: './home.component.html',
  styleUrl: './home.component.css',
})
export class HomeComponent {
  @ViewChild('conversationContainer')
  private conversationContainer?: ElementRef;
  private chatbotService = inject(ChatbotService);

  prompt: Prompt = { prompt: '' };
  conversation: Message[] = [];
  awaitingResponse: boolean = false;

  send_prompt() {
    this.awaitingResponse = true;
    this.conversation.push({ sender: 'user', message: this.prompt.prompt });
    this.chatbotService.send_prompt(this.prompt).subscribe({
      next: (response) => {
        this.conversation.push({ sender: 'bot', message: response.response });
        setTimeout(() => this.scrollToBottom(), 0);
        this.awaitingResponse = false;
      },
    });

    this.prompt = { prompt: '' };
    setTimeout(() => this.scrollToBottom(), 0);
  }

  scrollToBottom(): void {
    if (this.conversationContainer) {
      try {
        this.conversationContainer.nativeElement.scrollTop =
          this.conversationContainer.nativeElement.scrollHeight;
      } catch (err) {
        console.log('Scroll to bottom failed:', err);
      }
    }
  }
}
