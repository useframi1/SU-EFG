import { HttpClient } from '@angular/common/http';
import { Component, inject, OnInit } from '@angular/core';
import { RouterOutlet } from '@angular/router';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css',
})
export class AppComponent implements OnInit {
  http = inject(HttpClient);
  title: any = 'client';

  ngOnInit(): void {
    this.http.get('http://localhost:5001/').subscribe({
      next: (response) => (this.title = response),
      error: (error) => console.log(error),
      complete: () => console.log('Request has completed'),
    });
  }
}
